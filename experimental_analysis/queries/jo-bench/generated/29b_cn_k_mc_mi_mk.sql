SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info AS mi, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[us]' AND k.keyword = 'computer-animation' AND mi.info LIKE 'USA:%200%' AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;